function mysmosvm(filename,numruns)
    if strcmp(filename,'MNIST-13.csv')
        MNIST=csvread(filename);
        y=(MNIST(:,1)-2)'; % class 1,3 to class -1,1
        feature_size=length(MNIST(1,:))-1;
        data_size=length(MNIST(:,1));
        epsilon=10^(-2);
        kernel=MNIST(:,2:end)*MNIST(:,2:end)'; %used lots of times
        kyij=(y'*y).*kernel;
        to_print=[1:9,10:10:99,100:100:999,1000:1000:9999,10000:5000:70000]; % a matrix for print
        total_time=zeros(1,numruns);
        for i=1:numruns
            fprintf('%d run begins\n',i);
            result=[]; %result to print
            tStart=tic;
            b=0;
            C=0.0005;
            alpha=zeros(1,data_size); %initialzie the parameters
            w=zeros(1,feature_size);
            iter=0;
            numChanged=0;
            examineAll=1;
            E=-y; %initilize the error matrix
            unbound=alpha>0 & alpha<C; % unbound 1, otherwise 0
            while iter<=65000 % we wrun 60000 iterations
                for j=1:data_size
                    if unbound(j) || examineAll %only test data that is unbound if examineAll 0
                        i2=j;
                        alpha2=alpha(i2);
                        y2=y(i2);
                        E2=E(i2);
                        r2=E2*y2;
                        if (r2<-epsilon && alpha2<C )||(r2>epsilon && alpha2>0) %doesn't hold KKT condition
                            found=0; %record whether we have found the choice
                            if sum(unbound)>1 %more than one unbound point
                                [~,i1]=max(abs(E-E2).*unbound); %return the largest bound
                                %now test wether this one is possible
                                [a1,a2,condition]=takestep(i1,i2,E,y,alpha,kernel);
                                if condition
                                    found=1;
                                end      
                            end
                            
                            if ~found %second round for unbound element
                                start=randi(data_size); %starting point
                                for k=start:start+data_size-1
                                    temp_k=k;
                                    if k>data_size
                                        temp_k=k-data_size;
                                    end
                                    if unbound(temp_k)
                                        [a1,a2,condition]=takestep(temp_k,i2,E,y,alpha,kernel);
                                        if condition
                                            i1=temp_k;
                                            found=1;
                                            break;
                                        end
                                    end
                                end
                            end
                            
                            if ~found %third round for bound element
                                start=randi(data_size); %starting point
                                for k=start:start+data_size-1
                                    temp_k=k;
                                    if k>data_size
                                        temp_k=k-data_size;
                                    end
                                    if ~unbound(temp_k)
                                        [a1,a2,condition]=takestep(temp_k,i2,E,y,alpha,kernel);
                                        if condition
                                            i1=temp_k;
                                            found=1;
                                            break;
                                        end
                                    end
                                end
                            end
                            %now we have finished the second heuristic
                            if found %we have i1,i2
                                k11=kernel(i1,i1);
                                k12=kernel(i1,i2);
                                k22=kernel(i2,i2);         
                                alpha1=alpha(i1);
                                E1=E(i1);
                                y1=y(i1);
                                numChanged=numChanged+1;
                                %update b
                                if a1>0 && a1<C
                                    b=E1+y1*(a1-alpha1)*k11+y2*(a2-alpha2)*k12+b;
                                elseif a2>0 && a2<C
                                    b=E2+y1*(a1-alpha1)*k12+y2*(a2-alpha2)*k22+b;
                                else
                                    b1=E1+y1*(a1-alpha1)*k11+y2*(a2-alpha2)*k12+b;
                                    b2=E2+y1*(a1-alpha1)*k12+y2*(a2-alpha2)*k22+b;
                                    b=(b1+b2)/2;
                                end
                                %update w
                                w=w+y1*(a1-alpha1)*MNIST(i1,2:end)+y2*(a2-alpha2)*MNIST(i2,2:end);
                                %update E
                                E=(MNIST(:,2:end)*w')'-b-y;
                                %update alpha
                                alpha(i1)=a1;
                                alpha(i2)=a2;
                                %update unbound set
                                unbound(i1)=a1>0 & a1<C; %remember to update unbound
                                unbound(i2)=a2>0 & a2<C; 
                                %update objective function
                                %result=a1+a2-alpha1-alpha2;
                                %result=result+sum(y1*(alpha1-a1)*k(i1,:).*alpha.*y);
                                %result=result+sum(y2*(alpha2-a2)*k(i2,:).*alpha.*y);
                                %result=result-0.5*y1*y1*k(i1,i1)*alpha1*alpha1
                                %result=sum(alpha)-0.5*sum(sum(kyij.*(alpha'*alpha)));
                                iter=iter+1;  
                                if ismember(iter,to_print)
                                    result=[result,sum(alpha)-0.5*sum(sum(kyij.*(alpha'*alpha)))];                                                                        
                                end
                            end
                        end
                    end
                   
                end %end of i2 outer loop
                if examineAll
                    examineAll=0;
                elseif ~numChanged
                    examineAll=1;
                end
               
            end %end of while
            total_time(i)=toc(tStart); %record the time
            plot(to_print(1:length(result)),result);
            hold on;
        end
        xlabel('iteration')
        ylabel('objective function')
        fprintf('total mean runtime %f seconds, standard deviation %f seconds \n',[mean(total_time),std(total_time)]);
    end                
end